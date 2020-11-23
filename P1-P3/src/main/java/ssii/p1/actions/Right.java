/*Custom Vacuum World Example 
Copyright (C) 2015  Jorge J. Gomez-Sanz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */	
package ssii.p1.actions;

import ssii.p1.state.VacuumState;
import aima.core.agent.Action;

public class Right implements OOAction {

	private int aid;

	public Right(int aid){
		this.aid=aid;
	}

	public Right(){	}

	public boolean isNoOp() {	
		return false;
	}

	public Object perform(Object state) {
		VacuumState vs=((VacuumState)state).clone();
		vs.moveRight(aid);		
		return vs;
	}

	public boolean valid(Object state){
		VacuumState vs=((VacuumState)state);
		return vs.getWidth()>vs.getLocation(aid).getX()+1 
				&& vs.noAgentsSameLocation(
						new Location(vs.getLocation(aid).getX()+1, 
								vs.getLocation(aid).getY()));
	}

	public String toString(){ return "Right";};
}
